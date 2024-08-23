<h1>Image Colorization</h1>

This project is designed to convert grayscale images into color. The project is Dockerized for ease of deployment and portability.

<h2>Requirements</h2>

<ul>
  <li>Docker: Make sure Docker is installed and running on your local machine.</li>
</ul>

<h2>Steps to Run the Project</h2>

<p><b>1.Clone the Repository:</b> git clone https://github.com/RanaBetulKaya/GrayToColour_ImageProcessing.git</p>
<p><b>2.Download the Model:</b> Download the .pth model file from the provided Google Drive link and place it in the app/ directory.</p>
<p><b>3.Build the Docker Image:</b> Navigate to the root directory of the project and build the Docker image:<br> docker build -t pix2pix_image .</p>
<p><b>4.Run the Docker Container:</b> <br> Once the image is built, run the Docker container using the following command: <br> docker run --name pix2pix_container -p 8000:8000 pix2pix_image</p>
<p><b>5.Access the API:</b> <br> After the container is running, you can access the application at http://localhost:8000.</p>



