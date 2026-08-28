#!/usr/bin/env python3
"""
Transcription audio avec diarisation des locuteurs
Utilise faster-whisper pour la transcription et clustering pour la diarisation
"""

import os
import sys
import argparse
import subprocess
import tempfile
import wave
import struct
from pathlib import Path
import numpy as np
from faster_whisper import WhisperModel
import librosa
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings("ignore")

def convert_to_wav(input_path, output_path):
    """
    Convertit un fichier audio en WAV 16kHz mono en utilisant FFmpeg
    """
    cmd = [
        'ffmpeg',
        '-i', input_path,
        '-ar', '16000',  # 16 kHz sample rate
        '-ac', '1',      # mono
        '-c:a', 'pcm_s16le',  # 16-bit PCM
        '-y',  # overwrite output file
        output_path
    ]

    try:
        subprocess.run(cmd, check=True, capture_output=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Erreur lors de la conversion de {input_path}: {e}")
        return False


def transcribe_audio(wav_path, language="fr"):
    """
    Transcrit l'audio en utilisant faster-whisper avec timestamps au niveau des mots
    """
    print("Chargement du modèle faster-whisper...")
    model = WhisperModel("small", device="cpu", compute_type="int8")

    print("Transcription en cours...")
    segments, info = model.transcribe(
        wav_path,
        language=language,
        word_timestamps=True,
        vad_filter=True,  # Appliquer le filtre VAD pour supprimer les silences
        vad_parameters=dict(min_silence_duration_ms=500)
    )

    # Convertir les segments en liste pour une manipulation plus facile
    word_segments = []
    for segment in segments:
        for word in segment.words:
            word_segments.append({
                "word": word.word,
                "start": word.start,
                "end": word.end,
                "probability": word.probability
            })

    return word_segments, info.language


def extract_audio_features(wav_path, segments):
    """
    Extrait des caractéristiques MFCC pour chaque segment de mot
    """
    print("Chargement de l'audio pour l'extraction de caractéristiques...")
    y, sr = librosa.load(wav_path, sr=16000, mono=True)

    print("Extraction des caractéristiques MFCC...")
    features = []
    valid_segments = []

    for segment in segments:
        start_sample = int(segment["start"] * sr)
        end_sample = int(segment["end"] * sr)

        # S'assurer que les indices sont dans les limites
        start_sample = max(0, start_sample)
        end_sample = min(len(y), end_sample)

        if end_sample > start_sample:
            # Extraire le segment audio
            audio_segment = y[start_sample:end_sample]

            # Extraire les caractéristiques MFCC
            if len(audio_segment) > 0:
                mfccs = librosa.feature.mfcc(
                    y=audio_segment,
                    sr=sr,
                    n_mfcc=13,
                    hop_length=512,
                    n_fft=2048
                )
                # Prendre la moyenne sur le temps pour obtenir un vecteur caractéristique par segment
                mfccs_mean = np.mean(mfccs.T, axis=0)
                features.append(mfccs_mean)
                valid_segments.append(segment)

    if len(features) == 0:
        print("Aucune caractéristique extraite, retour des segments tels quels")
        return np.array([]), []

    return np.array(features), valid_segments


def assign_speakers(features, segments, num_speakers=None):
    """
    Attribue des locuteurs aux segments en utilisant le clustering
    """
    if len(features) == 0 or len(segments) == 0:
        return segments

    if num_speakers is None:
        # Essayer d'estimer le nombre de locuteurs (entre 2 et 4)
        # Pour simplifier, on utilise 2 ou 3 selon la durée totale
        # Dans une vraie implémentation, on pourrait utiliser des métriques comme le silhouette score
        num_speakers = 2  # Par défaut, on suppose 2 locuteurs
        # On pourrait ajuster basé sur d'autres facteurs si nécessaire

    print(f"Clustering en {num_speakers} locuteurs...")

    # Standardiser les caractéristiques
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    # Appliquer le clustering hiérarchique agglomératif
    clustering = AgglomerativeClustering(n_clusters=num_speakers)
    speaker_labels = clustering.fit_predict(features_scaled)

    # Attribuer les étiquettes de locuteur aux segments
    for i, segment in enumerate(segments):
        segment["speaker"] = int(speaker_labels[i])

    return segments


def merge_consecutive_same_speaker(segments):
    """
    Fusionne les segments consécutifs ayant le même locuteur
    """
    if not segments:
        return []

    merged = []
    current = segments[0].copy()
    current["text"] = current["word"]

    for segment in segments[1:]:
        # Si le locuteur est le même et que les segments sont proches (moins de 0.5s d'écart)
        if (segment["speaker"] == current["speaker"] and
            segment["start"] - current["end"] < 0.5):
            # Fusionner les textes
            current["end"] = segment["end"]
            current["text"] += " " + segment["word"]
        else:
            # Ajouter le segment actuel et commencer un nouveau
            current["text"] = current["text"].strip()
            merged.append(current)
            current = segment.copy()
            current["text"] = current["word"]

    # Ajouter le dernier segment
    current["text"] = current["text"].strip()
    merged.append(current)

    return merged


def assign_speaker_numbers(segments):
    """
    Attribue des numéros de locuteur lisibles (Locuteur 1, Locuteur 2, etc.)
    et s'assure qu'ils sont dans l'ordre d'apparition
    """
    if not segments:
        return segments

    # Créer un mapping basé sur l'ordre d'apparition du premier segment de chaque locuteur
    speaker_first_appearance = {}
    for segment in segments:
        speaker_id = segment["speaker"]
        if speaker_id not in speaker_first_appearance:
            speaker_first_appearance[speaker_id] = segment["start"]

    # Trier les locuteurs par leur première apparition
    sorted_speakers = sorted(speaker_first_appearance.items(), key=lambda x: x[1])

    # Créer le mapping
    speaker_mapping = {}
    for i, (speaker_id, _) in enumerate(sorted_speakers):
        speaker_mapping[speaker_id] = f"Locuteur {i + 1}"

    # Appliquer le mapping
    for segment in segments:
        segment["speaker_number"] = speaker_mapping[segment["speaker"]]

    return segments


def write_transcription_output(segments, output_path):
    """
    Écrit la transcription finale dans un fichier texte
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        for segment in segments:
            speaker = segment.get("speaker_number", "Locuteur inconnu")
            text = segment["text"].strip()
            if text:  # Ne pas écrire les lignes vides
                f.write(f"[{speaker}] {text}\n")


def process_single_file(input_path, output_dir=None):
    """
    Traite un seul fichier audio M4A
    """
    input_path = Path(input_path)

    if not input_path.exists():
        print(f"Erreur: Le fichier {input_path} n'existe pas.")
        return False

    # Déterminer le chemin de sortie
    if output_dir is None:
        output_dir = input_path.parent
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    output_filename = input_path.stem + "_transcription.txt"
    output_path = output_dir / output_filename

    print(f"Traitement de: {input_path}")
    print(f"Sortie vers: {output_path}")

    # Créer un répertoire temporaire pour les fichiers intermédiaires
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir = Path(temp_dir)
        wav_path = temp_dir / "audio.wav"

        # Étape 1: Conversion audio
        print("\n1. Conversion audio en WAV...")
        if not convert_to_wav(str(input_path), str(wav_path)):
            return False

        # Étape 2: Transcription
        print("\n2. Transcription avec faster-whisper...")
        transcription_result, language = transcribe_audio(str(wav_path), language="fr")
        print(f"Langue détectée: {language}")

        if not transcription_result:
            print("Aucun segment de transcription trouvé.")
            return False

        # Étape 3: Extraction de caractéristiques
        print("\n3. Extraction des caractéristiques audio...")
        features, valid_segments = extract_audio_features(str(wav_path), transcription_result)

        if len(valid_segments) == 0:
            print("Aucun segment valide après l'extraction de caractéristiques.")
            return False

        # Étape 4: Attribution des locuteurs par clustering
        print("\n4. Attribution des locuteurs par clustering...")
        segments_with_speakers = assign_speakers(features, valid_segments, num_speakers=2)

        # Étape 5: Fusion des segments consécutifs
        print("\n5. Fusion des segments consécutifs...")
        segments_merged = merge_consecutive_same_speaker(segments_with_speakers)

        # Étape 6: Attribution des numéros de locuteur
        print("\n6. Attribution des numéros de locuteur...")
        segments_final = assign_speaker_numbers(segments_merged)

        # Étape 7: Écriture de la sortie
        print("\n7. Écriture de la transcription finale...")
        write_transcription_output(segments_final, str(output_path))

    print(f"\nTranscription terminée! Résultat sauvegardé dans: {output_path}")
    return True


def process_directory(input_dir, output_dir=None):
    """
    Traite tous les fichiers M4A dans un répertoire
    """
    input_dir = Path(input_dir)

    if not input_dir.exists() or not input_dir.is_dir():
        print(f"Erreur: {input_dir} n'est pas un répertoire valide.")
        return False

    # Trouver tous les fichiers M4A (insensible à la casse)
    m4a_files = list(input_dir.rglob("*.m4a")) + list(input_dir.rglob("*.M4A"))

    if not m4a_files:
        print(f"Aucun fichier M4A trouvé dans {input_dir}")
        return False

    print(f"Trouvé {len(m4a_files)} fichier(s) M4A à traiter.")

    success_count = 0
    for m4a_file in m4a_files:
        if process_single_file(m4a_file, output_dir):
            success_count += 1
        print("-" * 50)

    print(f"\nTraitement terminé: {success_count}/{len(m4a_files)} fichiers traités avec succès.")
    return success_count > 0


def main():
    parser = argparse.ArgumentParser(
        description="Transcription audio avec diarisation des locuteurs"
    )
    parser.add_argument(
        "input",
        help="Chemin vers un fichier M4A ou un répertoire contenant des fichiers M4A"
    )
    parser.add_argument(
        "-o", "--output",
        help="Répertoire de sortie (par défaut: même répertoire que l'entrée)"
    )
    parser.add_argument(
        "--list-devices",
        action="store_true",
        help="Lister les appareils audio disponibles et quitter"
    )

    args = parser.parse_args()

    if args.list_devices:
        # Cette fonctionnalité pourrait être ajoutée si nécessaire
        print("Liste des appareils non implémentée dans cette version.")
        return

    input_path = Path(args.input)

    if not input_path.exists():
        print(f"Erreur: Le chemin spécifié n'existe pas: {args.input}")
        sys.exit(1)

    # Vérifier si c'est un fichier ou un répertoire
    if input_path.is_file():
        # Traiter un seul fichier
        success = process_single_file(
            input_path,
            output_dir=args.output
        )
    elif input_path.is_dir():
        # Traiter un répertoire
        success = process_directory(
            input_path,
            output_dir=args.output
        )
    else:
        print(f"Erreur: Le chemin spécifié n'est ni un fichier ni un répertoire: {args.input}")
        success = False

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()