FROM python3.9

COPY ["Software Implementations/", "."]
COPY ["requirements.txt", "."]

RUN pip install -U pip setuptools wheel
RUN pip install -r requirements.txt

RUN python -m spacy download en_core_web_trf




