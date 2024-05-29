# Use an official Python runtime as a parent image
FROM python:3.10

# Set the working directory
WORKDIR /usr/src/app

# Copy the requirements file into the container
COPY requirements.txt ./

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the working directory contents into the container
COPY . .

# Run app.py when the container launches
CMD ["chainlit", "run", "app.py"]
