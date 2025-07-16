# ### STAGE 1: Base Image ###
# Use an official lightweight Python image.
# Using -slim is a good practice for production to keep the image size down.
FROM python:3.11 as base

# ### STAGE 2: Setup Environment ###
# Set the working directory in the container.
WORKDIR /talk2bev

# Create a non-root user and switch to it.
# Running as a non-root user is a security best practice.
RUN useradd --create-home appuser
USER appuser

# ### STAGE 3: Install Dependencies ###
# Copy only the requirements file to leverage Docker's layer caching.
# This step will only be re-run if requirements.txt changes.
COPY --chown=appuser:appuser requirements.txt .

# Install the Python dependencies.
RUN pip install --no-cache-dir --upgrade pip -r requirements.txt
RUN wget https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.1/flash_attn-2.8.1+cu12torch2.5cxx11abiFALSE-cp311-cp311-linux_x86_64.whl
RUN pip install flash_attn-2.8.1+cu12torch2.5cxx11abiFALSE-cp311-cp311-linux_x86_64.whl
RUN cd data && wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth


# ### STAGE 4: Add Application Code ###
# Copy the rest of your application's code into the container.
COPY --chown=appuser:appuser . .

# ### STAGE 5: Run ###
# Expose the port your app runs on (e.g., 8000 for a web server).
# This is documentation; it doesn't actually publish the port.
EXPOSE 8000

# Define the command to run your application.
# Replace `your_app_file.py` with the entry point of your application.
CMD ["python", "your_app_file.py"]