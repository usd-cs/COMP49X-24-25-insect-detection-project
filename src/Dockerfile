# Use an official Python runtime as a parent image
FROM python:3.11

# Set the working directory in the container
WORKDIR /src

# Copy the requirements file into the container at /app
COPY src/requirements.txt requirements.txt 

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the current directory contents into the container at /app
COPY . .

# Define environment variable
ENV NAME InsectDetection

# Run app.py when the container launches
CMD ["python", "src/simulator.py"]