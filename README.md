# Transcription Audio avec Diarisation des Locuteurs, les données restent locales

Ce projet fournit un outil pour transcrire LOCALEMENT des fichiers audio M4A en différenciant les voix des locuteurs (diarisation). Il utilise :
- **FFmpeg** pour la conversion audio
- **Faster-whisper** pour la transcription avec timestamps au niveau des mots
- **Clustering de caractéristiques MFCC** pour la diarisation des locuteurs

## Prérequis

- Python 3.8+
- FFmpeg (doit être dans le PATH)

## Installation

1. Clonez ce dépôt ou copiez les fichiers dans un répertoire
2. Installez les dépendances Python :
   ```bash
   pip install -r requirements.txt
   ```

## Utilisation

### Transcrire un seul fichier
```bash
python transcriber.py chemin/vers/fichier.m4a
```

### Transcrire un répertoire entier
```bash
python transcriber.py chemin/vers/repertoire/
```

### Spécifier un répertoire de sortie
```bash
python transcriber.py chemin/vers/fichier.m4a -o chemin/vers/sortie/
```

### Options avancées
```bash
# Choisir un modèle Whisper différent (tiny, base, small, medium, large)
python transcriber.py fichier.m4a --model medium

# Spécifier le seuil de distance pour le clustering
python transcriber.py fichier.m4a --threshold 1.5

# Changer la langue de transcription (défaut: fr)
python transcriber.py fichier.m4a --language en

# Changer le format de sortie (txt, json, srt)
python transcriber.py fichier.m4a --format srt
```

## Format de sortie

La transcription peut être enregistrée dans trois formats différents :

### Texte simple (par défaut)
Fichier `[nom_fichier_source]_transcription.txt` :
```
[Locuteur 1] Bonjour, comment allez-vous aujourd'hui ?
[Locuteur 2] Je vais très bien, merci ! Et toi ?
[Locuteur 1] Je suis un peu fatigué mais ça va.
```

### JSON
Fichier `[nom_fichier_source]_transcription.json` :
```json
[
  {
    "speaker": "Locuteur 1",
    "text": "Bonjour, comment allez-vous aujourd'hui ?",
    "start_time": 0.0,
    "end_time": 3.5
  },
  {
    "speaker": "Locuteur 2",
    "text": "Je vais très bien, merci ! Et toi ?",
    "start_time": 3.5,
    "end_time": 7.2
  }
]
```

### SRT (SubRip Subtitle)
Fichier `[nom_fichier_source]_transcription.srt` :
```
1
00:00:00,000 --> 00:00:03,500
[Locuteur 1] Bonjour, comment allez-vous aujourd'hui ?

2
00:00:03,500 --> 00:00:07,200
[Locuteur 2] Je vais très bien, merci ! Et toi ?
```

## Fonctionnement

1. **Conversion audio** : Le fichier M4A est converti en WAV 16kHz mono avec FFmpeg
2. **Transcription** : Faster-whisper transcrit l'audio avec des timestamps au niveau des mots
3. **Extraction de caractéristiques** : Des caractéristiques MFCC sont extraites pour chaque segment de mot
4. **Diarisation** : Un clustering hiérarchique agglomératif groupe les segments par locuteur (en utilisant soit un nombre fixe de locuteurs, soit un seuil de distance)
5. **Post-traitement** : Les segments consécutifs du même locuteur sont fusionnés
6. **Sortie** : Le résultat est écrit dans le format spécifié avec des étiquettes de locuteur lisibles

## Notes importantes

- Le modèle faster-whisper "small" est utilisé par défaut pour un bon équilibre entre précision et performance
- Sur un CPU moyen, attendez-vous à ce que le traitement prenne environ 1-2x la durée de l'audio
- Aucune authentification Hugging Face n'est requise contrairement aux solutions basées sur pyannote.audio
- La solution est entièrement open-source et fonctionne localement sans envoyer de données à des serveurs externes
<<<<<<< HEAD
- Le clustering peut être contrôlé soit par `--speakers` (nombre de locuteurs) soit par `--threshold` (seuil de distance), mais pas les deux en même temps

