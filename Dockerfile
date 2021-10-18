FROM intel/intel-optimized-pytorch
WORKDIR /app
COPY . /app
RUN pip install --no-cache-dir lightning-flash[text] sklearn fire datasets==1.12.1
# ENTRYPOINT ["python", "train.py"] 

