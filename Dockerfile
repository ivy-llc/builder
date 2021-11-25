FROM ivydl/ivy:latest

# Install Ivy
RUN rm -rf ivy && \
    git clone https://github.com/ivy-dl/ivy && \
    cd ivy && \
    cat requirements.txt | grep -v "ivy-" | pip3 install --no-cache-dir -r /dev/stdin && \
    cat optional.txt | grep -v "ivy-" | pip3 install --no-cache-dir -r /dev/stdin && \
    python3 setup.py develop --no-deps

COPY requirements.txt /
RUN cat requirements.txt | grep -v "ivy-" | pip3 install --no-cache-dir -r /dev/stdin

COPY optional.txt /
RUN cat optional.txt | grep -v "ivy-" | pip3 install --no-cache-dir -r /dev/stdin

RUN python3 test_dependencies.py -fp requirements.txt,optional.txt && \
    rm -rf requirements.txt && \
    rm -rf optional.txt

RUN mkdir ivy_builder
WORKDIR /ivy_builder