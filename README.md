Download Link: https://assignmentchef.com/product/solved-coms6998-homework-5
<br>
<h1>Problem 1 – <em>SSD, ONNX model, Visualization, Inferencing    </em></h1>

In this problem we will be inferencing SSD ONNX model using ONNX Runtime Server. You will follow the github repo and ONNX tutorials (links provided below). You will start with a pretrained Pytorch SSD model and retrain it for your target categories. Then you will convert this Pytorch model to ONNX and deploy it on ONNX runtime server for inferencing.

<ol>

 <li>Download pretrained pytorch MobilenetV1 SSD and test it locally using Pascal VOC 2007 dataset. Show the test accuracy for the 20 classes. (4)</li>

 <li>Select any two related categories from Google Open Images dataset and finetune the pretrained SSD model. Examples include, Aircraft and Aeroplane, Handgun and Shotgun. You can use py script provided at the github to download the data. For finetuning you can use the same parameters as in the tutorial below. Compute the accuracy of the test data for these categories before and after finetuning. (5+5)</li>

 <li>Convert the Pytorch model to ONNX format and save it. (4)</li>

 <li>Visualize the model using net drawer tool. Compile the model using embed_docstring flag and show the visualization output. Also show doc string (stack trace for PyTorch) for different types of nodes.</li>

</ol>

(6)

<ol start="5">

 <li>Deploy the ONNX model on ONNX runtime (ORT) server. You need to set up the environment following steps listed in the tutorial. Then you need make HTTP request to the ORT server. Test the inferencing set-up using 1 image from each of the two selected categories. (6)</li>

 <li>Parse the response message from the ORT server and annotate the two images. Show inferencing output (bounding boxes with labels) for the two images. (5)</li>

</ol>

For part 1, 2, and 3, refer to the steps in the github repo. For part 4 refer to ONNX tutorial on visualizing and for 5 and 6 refer to ONNX tutorial on inferencing.

<em>References </em>• Github repo. Shot MultiBox Detector Implementation in Pytorch. Available at <a href="https://github.com/qfgaohao/pytorch-ssd">https://github.com/qfgaohao/pytorch-ssd </a>• ONNX tutorial. Visualizing an ONNX Model.

Available at <a href="https://github.com/onnx/tutorials/blob/master/tutorials/VisualizingAModel.md">https://github.com/onnx/tutorials/blob/master/tutorials/VisualizingAModel.md </a>• ONNX tutorial. Inferencing SSD ONNX model using ONNX Runtime Server.

Available at <a href="https://github.com/onnx/tutorials/blob/master/tutorials/OnnxRuntimeServerSSDModel.ipynb">https://github.com/onnx/tutorials/blob/master/tutorials/OnnxRuntimeServerSSDModel.ipynb </a>• Google. Open Images Dataset V5 + Extensions.

Available at <a href="https://storage.googleapis.com/openimages/web/index.html">https://storage.googleapis.com/openimages/web/index.html </a>• The PASCAL Visual Object Classes Challenge 2007.

Available at <a href="http://host.robots.ox.ac.uk/pascal/VOC/voc2007/#testdata">http://host.robots.ox.ac.uk/pascal/VOC/voc2007/</a>

<h1>Problem 2 – <em>ML Cloud Platforms      </em></h1>

In this question you will analyze different ML cloud platforms and compare their service offerings. In particular, you will consider ML cloud offerings from IBM, Google, Microsoft, and Amazon and compare them on the basis of following criteria:

<ol>

 <li>Frameworks: DL framework(s) supported and their version. (4)</li>

</ol>

<em>Here we are referring to machine learning platforms which have their own inbuilt images for different frameworks.</em>

<ol start="2">

 <li>Compute units: type(s) of compute units offered, i.e., GPU types. (2)</li>

 <li>Model lifecycle management: tools supported to manage ML model lifecycle. (2)</li>

 <li>Monitoring: availability of application logs and resource (GPU, CPU, memory) usage monitoring data to the user. (2)</li>

 <li>Visualization during training: performance metrics like accuracy and throughput (2)</li>

 <li>Elastic Scaling: support for elastic scaling compute resources of an ongoing job. (2)</li>

 <li>Training job description: training job description file format. Show how the same training job is specified in different ML platforms. Identify similar fields in the training job file for the 4 ML platforms through an example. (6)</li>

</ol>

<h2><strong>Problem 3 – </strong>Kubeflow, MiniKF, Kale</h2>

In this problem we will follow Kubeflow-Kale codelab (link below). You will follow the steps as outlined in the codelab to install Kubeflow with MiniKF, convert a Jupyter Notebook to Kubeflow Pipelines, and run Kubeflow Pipelines from inside a Notebook. <strong>For each step below you need to show the commands executed, terminal output, and screenshot of visual output (if any). You also need to give a new name to your GCP project and any resource instance you create, e.g., put your initial in the name string.</strong>

<ol>

 <li><em>Setting up the environment and installing MiniKF</em>: Follow the steps in the codelab to:

  <ul>

   <li>Set up a GCP project. (2)</li>

   <li>Install MiniKF and deploy your MinKF instance. (3)</li>

   <li>Login to MiniKF, Kubeflow, and Rok. (3)</li>

  </ul></li>

 <li><em>Run a Pipeline from inside your Notebook</em>: Follow the steps in the codelab to:

  <ul>

   <li>Create a notebook server. (3)</li>

   <li>Download and run the notebook: We will be using <strong>pytorch-classification </strong>notbeook from the example repo. <em>Note that the codelab uses a different example from the repo (titanic dataset ml.ipynb)</em>. (4)</li>

   <li>Convert your notebook to a Kubeflow Pipeline: Enable Kale and then compile and run the pipeline from Kale Deployment Panel. Show output from each of the 5 steps of the pipeline (5)</li>

   <li>Show snapshots of ”Graph” and ”Run output” of the experiment. (4)</li>

   <li><em>Cleanup: </em>Destroy the MiniKF VM. (1)</li>

  </ul></li>

</ol>

<em>References </em>• Codelab. From Notebook to Kubeflow Pipelines with MiniKF and Kale.

Available at <a href="https://codelabs.developers.google.com/codelabs/cloud-kubeflow-minikf-kale">https://codelabs.developers.google.com/codelabs/cloud-kubeflow-minikf-kale</a>

<ul>

 <li><a href="https://github.com/kubeflow-kale/examples">https://github.com/kubeflow-kale/examples</a></li>

</ul>

<h2><strong>Problem 4 – </strong>Deep Reinforcement Learning</h2>

This question is based on Deep RL concepts discussed in Lecture 8. You need to refer to the papers by Mnih et al., Nair et al., and Horgan et al. to answer this question. All papers are linked below.

<ol>

 <li>Explain the difference between episodic and continuous tasks? Given an example of each. (2)</li>

 <li>What do the terms exploration and exploitation mean in RL ? Why do the actors employ -greedy policy for selecting actions at each step? Should remain fixed or follow a schedule during Deep RL training ? How does the value of <em> </em>help balance exploration and exploitation during training. (1+1+1+1)</li>

 <li>How is the Deep Q-Learning algorithm different from Q-learning ? You will follow the steps of Deep Q-Learning algorithm in Mnih et al. (2013) page 5, and explain each step in your own words. (3)</li>

 <li>What is the benefit of having a target Q-network ? (3)</li>

 <li>How does experience replay help in efficient Q-learning ? (3)</li>

 <li>What is prioritized experience replay ? (2)</li>

 <li>Compare and contrast GORILA (General Reinforcement Learning Architecture) and Ape-X architecture. Provide three similarities and three differences. (3)</li>

</ol>