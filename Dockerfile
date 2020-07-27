FROM python:3.8
COPY PIE /app/PIE
COPY readme.md /app/readme.md
COPY setup.py /app/setup.py
WORKDIR /app
RUN pip install /app
RUN rm -Rf /app
WORKDIR /
