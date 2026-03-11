 # SPMe+: A Spatially-Resolved SPM for LIB/LMB

## Installation

1. Clone this repo to your machine.
2. Install the UV package manager:
    ```shell
   # On macOS and Linux:
   curl -LsSf https://astral.sh/uv/install.sh | sh
   # On Windows:
   powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
   ```
3. Navigate to the project's root directory and install project dependencies with UV:
   ```shell
   uv sync
   ```