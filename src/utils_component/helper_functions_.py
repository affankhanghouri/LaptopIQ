
def categorize_cpu(cpu):
    cpu = str(cpu).lower()  # Convert to lowercase for consistency
    if "i5" in cpu:
        return 0  # Label for Intel i5
    elif "i7" in cpu:
        return 1  # Label for Intel i7
    elif "intel" in cpu:
        return 2  # Label for other Intel CPUs (Celeron, Pentium, etc.)
    elif "amd" in cpu:
        return 3  # Label for AMD processors
    else:
        return 4  # Optional: For any other unknown CPUs




# Function to extract memory values
def extract_memory(memory, storage_type):
    memory = memory.lower()  # Convert to lowercase for consistency
    if storage_type in memory:
        # Extract the numeric value before the storage type
        parts = memory.split()
        for i, part in enumerate(parts):
            if storage_type in part:
                try:
                    return int(parts[i-1].replace('gb', '').replace('tb', '000'))  # Convert TB to GB
                except:
                    return 0
    return 0



def categorize_gpu(gpu):
    gpu = str(gpu).lower()
    if "intel" in gpu:
        return "intel"
    elif "amd" in gpu:
        return "Amd"
    elif "nvidia" in gpu:
        return "Nividia"
    else:
        return "other"


def categorize_opsys(ops):
    op=str(ops).lower()

    if "windows 10" in op:
        return "Windows 10"
    elif "windows 7"in op:
        return "Windows 7"

    elif "linux" in op:
        return "linux"

    elif "macos" in op:
        return "Macos" 
    else:
        return "other"


