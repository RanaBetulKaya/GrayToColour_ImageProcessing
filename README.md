<h1>Image Colorization</h1>

This project is designed to convert grayscale images into color. The project is Dockerized for ease of deployment and portability.

<h2>Requirements</h2>

<ul>
  <li>Docker: Make sure Docker is installed and running on your local machine.</li>
</ul>

<h2>Steps to Run the Project</h2>

<p><b>1.Clone the Repository:</b> 
<ul>
  <li> git clone https://github.com/RanaBetulKaya/GrayToColour_ImageProcessing.git</li>
</ul>
</p>
<p><b>2.Download the Model:</b> 
<ul>
  <li> Download the .pth model file from the provided Google Drive link and place it in the app/ directory.</li>
</ul>
</p>
<p><b>3.Build the Docker Image:</b> 
<ul>
  <li> Navigate to the root directory of the project and build the Docker image:</li>
<br> docker build -t pix2pix_image .
</ul>
</p>
<p><b>4.Run the Docker Container:</b> 
<ul> 
  <li>Once the image is built, run the Docker container using the following command:</li> <br> docker run --name pix2pix_container -p 8000:8000 pix2pix_image
</ul>  
</p>
<p><b>5.Access the API:</b> <ul>
<li> After the container is running, you can access the application at http://localhost:8000.</li></ul></p>



