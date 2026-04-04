# Medical-Chatbot

# How to run?
### STEPS:
 
Clone the repository

''' bash
git clone https://github.com/vidyazade/Medical-Chatbot.git
'''
### STEP 01- Create a conda environment after opening the repository

''' bash
conda create -n medibot python=3.11.9 -y
'''

'''bash
conda activate medibot
'''

### STEP 02- Install the requirements
'''bash
pip install -r requirements.txt
'''

### Create a .env file in the root directory and add your Pinecone & openai credentials as follows:
PINECONE_API_KEY = "xxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
OPENAI_API_KEY = "xxxxxxxxxxxxxxxxxxxxxxxxxxxxx"

# run the following command to store embeddings to pinecone
python store_index.py

# Finally run the following command
python app.py

Now,
```bash
open up localhost:
```

### Techstack Used

-Python
-LangChain
-Flask
-GPT
-Pinecone

# AWS-CICD-Deployment-with-Github-Actions

## 1. Login to AWS console.
## 2. Create IAM user for deployment
### #with specific access

1. EC2 access : It is virtual machine

2. ECR: Elastic Container registry to save your docker image in aws


#Description: About the deployment

1. Build docker image of the source code

2. Push your docker image to ECR

3. Launch Your EC2 

4. Pull Your image from ECR in EC2

5. Lauch your docker image in EC2

#Policy:

1. AmazonEC2ContainerRegistryFullAccess

2. AmazonEC2FullAccess

## 3. Create ECR repo to store/save docker image
