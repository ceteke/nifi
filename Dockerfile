FROM pytorch/pytorch:2.7.0-cuda11.8-cudnn9-devel

COPY requirements.txt /requirements.txt
WORKDIR /

RUN pip install -r requirements.txt
RUN pip install torch-scatter -f https://data.pyg.org/whl/torch-2.7.0+cu118.html

ENTRYPOINT ["tail", "-f", "/dev/null"]