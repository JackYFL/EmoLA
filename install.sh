# Install main packages
conda create -n emola python=3.10 -y
conda activate emola
pip install --upgrade pip  # enable PEP 660 support
pip install -e .

# Install extra libraries
pip install wandb==0.15.12
pip install deepspeed==0.12.6
pip install flash-attn --no-build-isolation
pip install rouge
pip install ipdb