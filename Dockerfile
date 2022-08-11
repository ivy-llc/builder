FROM unifyai/ivy:latest

# Install Ivy
RUN rm -rf ivy && \
    git clone https://github.com/unifyai/ivy && \
    cd ivy && \
    cat requirements.txt | grep -v "ivy-" | pip3 install --no-cache-dir -r /dev/stdin && \
    cat optional.txt | grep -v "ivy-" | pip3 install --no-cache-dir -r /dev/stdin && \
    python3 -m pip install --user -e .

# Install Ivy Builder
RUN git clone https://github.com/unifyai/builder && \
    cd builder && \
    cat requirements.txt | grep -v "ivy-" | pip3 install --no-cache-dir -r /dev/stdin && \
    cat optional.txt | grep -v "ivy-" | pip3 install --no-cache-dir -r /dev/stdin && \
    python3 -m pip install --user -e .

COPY requirements.txt /
RUN cat requirements.txt | grep -v "ivy-" | pip3 install --no-cache-dir -r /dev/stdin

COPY optional.txt /
RUN cat optional.txt | grep -v "ivy-" | pip3 install --no-cache-dir -r /dev/stdin

# RUN python3 test_dependencies.py -fp requirements.txt,optional.txt && \
#     rm -rf requirements.txt && \
#     rm -rf optional.txt

WORKDIR /builder