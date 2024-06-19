<h1>Neural Style Transfer with Streamlit</h1>
<h2>Project Description</h2>
    <p>
        This repository showcases an application of Neural Style Transfer using a VGG-19 model pretrained on ImageNet. 
        The app enables users to merge the artistic style of one image with the content of another, creating unique stylized images. 
        Powered by Streamlit, the app provides an easy-to-use web interface for seamless user interaction.
    </p>
    <h2>Setup Guide</h2>
    <p>Follow these steps to set up the environment and run the application:</p>
    <ol>
        <li>
            <strong>Clone the Repository</strong>
        </li>
        <li>
            <strong>Install Required Packages:</strong>
            <pre><code>pip install -r requirements.txt
            </code></pre>
        </li>
        <li>
            <strong>Launch the Application:</strong>
            <pre><code>streamlit run app.py
            </code></pre>
        </li>
    </ol>
    <h2>How to Use</h2>
    <ol>
        <li>
            <strong>Launch Streamlit:</strong>
            <p>Start the app with the command:</p>
            <pre><code>streamlit run app.py
            </code></pre>
        </li>
        <li>
            <strong>Upload Your Images:</strong>
            <ul>
                <li>Select a style image (e.g., a piece of artwork).</li>
                <li>Select a content image (e.g., a photo).</li>
            </ul>
        </li>
        <li>
            <strong>Create a Stylized Image:</strong>
            <ul>
                <li>Click the "Generate Image" button.</li>
                <li>Wait for the image processing to complete (this may take a few moments).</li>
                <li>View and download your newly stylized image.</li>
            </ul>
        </li>
    </ol>
    <h2>Prerequisites</h2>
    <p>Ensure you have the following libraries installed:</p>
    <ul>
        <li><code>streamlit</code>: For the web application interface.</li>
        <li><code>torch</code>: For the neural network operations.</li>
        <li><code>torchvision</code>: For accessing the pretrained VGG-19 model.</li>
        <li><code>PIL</code>: For image handling.</li>
        <li><code>matplotlib</code>: For image visualization (optional, mainly for development purposes).</li>
    </ul>
