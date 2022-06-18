# Install conda.
MINICONDA_INSTALLER_SCRIPT=Miniconda3-py39_4.10.3-Linux-x86_64.sh

if ! test -f "miniconda.sh"; then
    curl -sLo ~/miniconda.sh https://repo.continuum.io/miniconda/$MINICONDA_INSTALLER_SCRIPT
fi

chmod +x ~/miniconda.sh \
    && ~/miniconda.sh -b -p ~/miniconda \
    && rm ~/miniconda.sh \
    && echo ". ~/miniconda/etc/profile.d/conda.sh" >> ~/.bashrc \
    && echo "conda activate base" >> ~/.bashrc \
    && source ~/miniconda/etc/profile.d/conda.sh


# Update conda
conda install --channel defaults conda python=3.9 --yes
conda update --channel defaults --all --yes

# Install dependencies
conda env update -n base -f gpu_environment.yml \
    && conda clean -ya \
    && bash --login