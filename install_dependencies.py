import sys
import subprocess

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

packages = [
    "aiohttp",
    "beautifulsoup4",
    "nltk",
    "numpy",
    "scikit-learn",
    "torch",
    "transformers",
    "scipy"
]

print("Installation des packages nécessaires...")

for package in packages:
    print(f"Installation de {package}...")
    try:
        install(package)
        print(f"{package} installé avec succès.")
    except Exception as e:
        print(f"Erreur lors de l'installation de {package}: {str(e)}")

print("Installation terminée.")

import nltk

nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')

print("Configuration terminée. Vous pouvez maintenant exécuter le crawler.")