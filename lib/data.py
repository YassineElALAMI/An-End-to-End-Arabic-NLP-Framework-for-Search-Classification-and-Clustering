import zipfile
import os
import sys

JAR_PATH = "lib/arabic-stemmer.jar"

def inspect_jar(jar_path):
    if not os.path.exists(jar_path):
        print("JAR file not found:", jar_path)
        sys.exit(1)

    with zipfile.ZipFile(jar_path, 'r') as jar:
        print("=== JAR CONTENT ===\n")
        for name in jar.namelist():
            print(name)

        print("\n=== POSSIBLE STEMMER CLASSES ===\n")
        for name in jar.namelist():
            if name.endswith(".class") and ("stem" in name.lower() or "arab" in name.lower()):
                print(name)

if __name__ == "__main__":
    inspect_jar(JAR_PATH)
