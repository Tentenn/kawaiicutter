
import subprocess

# Read the requirements from the file
with open("requirements.txt", "r") as file:
    lines = file.readlines()

successful_packages = []
failed_packages = []

for line in lines:
    line = line.strip()  # Remove extra spaces and newline characters
    if line:  # Check if line is not empty
        try:
            # Try to install the package
            subprocess.check_call(["pip", "install", line])
            successful_packages.append(line)
        except subprocess.CalledProcessError:
            failed_packages.append(line)

# Write successful packages to successful_requirements.txt
with open("successful_requirements.txt", "w") as file:
    for pkg in successful_packages:
        file.write(pkg + "\n")

# Write failed packages to failed_requirements.txt
with open("failed_requirements.txt", "w") as file:
    for pkg in failed_packages:
        file.write(pkg + "\n")

print("Installation complete. Check successful_requirements.txt and failed_requirements.txt for details.")
