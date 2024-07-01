# Use an official Python runtime as a parent image
FROM python:3.10

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Create a virtual environment
RUN python -m venv venv

# Activate the virtual environment and install dependencies
RUN venv/Scripts/activate && pip install --upgrade pip && pip install -r requirements.txt"

# Make sure the virtualenv binaries are used
ENV PATH="/venv/bin:$PATH"

# Specify the command to run your application
CMD ["python", "app.py"]