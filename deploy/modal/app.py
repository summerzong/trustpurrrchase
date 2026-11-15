"""Deploy the existing TrustPurrrchase server on Modal."""
import os
from pathlib import Path
import subprocess

import modal

remote_demo = "/app/demo"
catalog_dir = "/app/catalog"
app = modal.App("trustpurrrchase")
catalog = modal.Volume.from_name("trustpurrrchase-catalog", create_if_missing=True)
image = None
if modal.is_local():
    demo_dir = Path(__file__).resolve().parents[2] / "demo"
    image = (
        modal.Image.debian_slim(python_version="3.11")
        .pip_install_from_requirements(str(demo_dir / "requirements.txt"))
        .add_local_dir(
            demo_dir,
            remote_path=remote_demo,
            ignore=[".env", ".env.*", ".cache", "out", "**/__pycache__"],
        )
    )


@app.function(
    image=image,
    secrets=[modal.Secret.from_name("trustpurrrchase-api", required_keys=["AIMO_API_KEY", "GEMINI_API_KEY"])],
    volumes={catalog_dir: catalog},
    max_containers=1,
    memory=768,
    scaledown_window=180,
    timeout=180,
)
@modal.web_server(7860, startup_timeout=180, label="demo")
def web():
    env = dict(os.environ, HOST="0.0.0.0", PORT="7860", HOME_UI="chat",
               AIMO_DATA_DIR=catalog_dir, TZ="Europe/Helsinki", PYTHONUNBUFFERED="1")
    catalog_files = ("product_data.json", "delivery_reliability.csv", "carbon_footprint.csv")
    if not all(Path(catalog_dir, "valio_aimo_junction26_" + name).exists() for name in catalog_files):
        subprocess.run(["python", "fetch_data.py"], cwd=remote_demo, env=env, check=True)
        catalog.commit()
    subprocess.Popen(["python", "server.py"], cwd=remote_demo, env=env)
