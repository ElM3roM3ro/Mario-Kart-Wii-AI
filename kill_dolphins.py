import subprocess

def close_dolphin_instances():
    """
    Closes all running Dolphin processes using the Windows taskkill command.
    """
    try:
        result = subprocess.run(
            'taskkill /F /IM Dolphin.exe',
            check=True,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        print("Dolphin instances closed successfully:")
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print("Error closing Dolphin instances:")
        print(e.stderr)

if __name__ == "__main__":
    close_dolphin_instances()
