# Transcription Audio avec Diarisation des Locuteurs, les données restent locales

Ce projet fournit un outil pour transcrire LOCALEMENT des fichiers audio M4A en différenciant les voix des locuteurs (diarisation). Il utilise :
- **FFmpeg** pour la conversion audio
- **Faster-whisper** (modèle small) pour la transcription avec timestamps au niveau des mots
- **Clustering de caractéristiques MFCC** pour la diarisation des locuteurs

## Prérequis

- Python 3.8+
- FFmpeg (doit être dans le PATH)
- Aucune dépendance supplémentaire complexe requise

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

## Format de sortie

La transcription est enregistrée dans un fichier texte nommé `[nom_fichier_source]_transcription.txt` avec le format suivant :
```
[Locuteur 1] Bonjour, comment allez-vous aujourd'hui ?
[Locuteur 2] Je vais très bien, merci ! Et toi ?
[Locuteur 1] Je suis un peu fatigué mais ça va.
```

## Fonctionnement

1. **Conversion audio** : Le fichier M4A est converti en WAV 16kHz mono avec FFmpeg
2. **Transcription** : Faster-whisper transcrit l'audio avec des timestamps au niveau des mots
3. **Extraction de caractéristiques** : Des caractéristiques MFCC sont extraites pour chaque segment de mot
4. **Diarisation** : Un clustering hiérarchique agglomératif groupe les segments par locuteur
5. **Post-traitement** : Les segments consécutifs du même locuteur sont fusionnés
6. **Sortie** : Le résultat est écrit dans un fichier texte avec des étiquettes de locuteur lisibles

## Notes importantes

- Le modèle faster-whisper "small" est utilisé par défaut pour un bon équilibre entre précision et performance
- Sur un CPU moyen, attendez-vous à ce que le traitement prenne environ 1-2x la durée de l'audio
- Aucune authentification Hugging Face n'est requise contrairement aux solutions basées sur pyannote.audio
- La solution est entièrement open-source et fonctionne localement sans envoyer de données à des serveurs externes
