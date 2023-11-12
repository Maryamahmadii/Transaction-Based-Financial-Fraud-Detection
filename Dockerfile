
# Base image
FROM continuumio/miniconda3:22.11.1

# Install essential packages
RUN pip install --no-cache-dir --upgrade pip pip-tools setuptools

# Copy the requirements file and install the required pip packages
COPY requirements.txt .
RUN pip install -r requirements.txt


RUN mkdir /home/workdir
WORKDIR /home/workdir


EXPOSE 8888
ENTRYPOINT ["jupyter", "lab", "--ip='0.0.0.0'", "--port=8888", "--no-browser", "--allow-root"]
