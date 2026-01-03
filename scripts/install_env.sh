conda create -n cad -c conda-forge python=3.11 -y
conda activate cad
conda install -c conda-forge ipython pythonocc-core=7.9.0 -y
pip3 install -U xformers==0.0.28.post3
pip install torch-cluster -f https://data.pyg.org/whl/torch-2.5.1+cu124.html
pip install packaging
pip install pymeshlab jaxtyping boto3 h5py
pip install trimesh beartype lightning safetensors open3d 
pip install omegaconf sageattention triton scikit-image transformers gpustat
pip install wandb pudb
pip install libigl h5py
pip install diffusers
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124
pip install opencv-python
pip install jsonargparse
cd csrc/rotary && python setup.py install 
cd ../layer_norm && python setup.py install 
cd ../xentropy && python setup.py install 
