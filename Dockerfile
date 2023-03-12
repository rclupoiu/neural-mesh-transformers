from python:3.10

RUN pip install --upgrade pip
RUN pip install torch torchvision torchaudio
RUN pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv torch_geometric -f https://data.pyg.org/whl/torch-1.13.0+cu117.html
RUN pip install numpy
RUN pip install scipy
RUN pip install pandas
RUN pip install scikit-learn scikit-image
RUN pip install umap-learn
RUN pip install seaborn
RUN pip install jupyter
RUN pip install tqdm
RUN pip install matplotlib
RUN pip install ceviche