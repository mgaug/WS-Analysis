# Dockerfile
FROM python:3.9 

# Set the working directory inside the container
WORKDIR /ws

# Copy any local files to the container
COPY . /ws

# Set the PYTHONPATH needed for scripts running in Notebooks directory
ENV PYTHONPATH "${PYTHONPATH}:/ws"

# Update pip and install Jupyter
RUN pip install --upgrade pip && \
    pip install -r requirements.txt 

RUN chmod +x /ws/run_docker.sh

# Expose the Jupyter port
EXPOSE 8888

# Run the Jupyter notebook when the container starts
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--NotebookApp.default_url=/ws/WS_jupyter.ipynb"]

CMD ["python","mess.py"]
#RUN tail -f /dev/null
