FROM ivydl/ivy:latest

# Install Ivy
RUN git clone https://github.com/ivy-dl/ivy && \
    cd ivy && \
    cat requirements.txt | grep -v "ivy-" | pip3 install --no-cache-dir -r /dev/stdin && \
    cat optional.txt | grep -v "ivy-" | pip3 install --no-cache-dir -r /dev/stdin && \
    python3 setup.py install && \
    cd ..

RUN mkdir ivy_builder
WORKDIR /ivy_builder

COPY requirements.txt /ivy_builder
RUN cat requirements.txt | grep -v "ivy-" | pip3 install --no-cache-dir -r /dev/stdin && \
    rm -rf requirements.txt