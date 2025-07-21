FROM jupyter/minimal-notebook:python-3.10

# Set working directory inside the container
WORKDIR /home/jovyan/work

# Copy requirements file separately for better Docker caching
COPY ./requirements.txt /home/jovyan/work/requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project files
COPY . .

# Expose Jupyter Notebook port
EXPOSE 8888

# Start Jupyter without token/password and set notebook directory
CMD ["start-notebook.sh", "--NotebookApp.token=", "--NotebookApp.password=", "--NotebookApp.allow_origin='*'", "--NotebookApp.notebook_dir=/home/jovyan/work"]
