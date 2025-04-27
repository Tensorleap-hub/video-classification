# 🎥 Action Recognition Demo with Tensorleap

This guide will help you run the action recognition demo using Tensorleap.

Follow the steps below to set up the environment, run tests locally, and deploy the project to Tensorleap's platform.

---

## 🚀 Getting Started

### 1. Clone the Repository

```
git clone git@github.com:Tensorleap-hub/video-classification.git
cd video-classification
```
### 2. Install Poetry
```
pip install poetry
```
### 3. Install Dependencies
```
poetry env use /path/to/python3.8
poetry install
```
### 4. Run a local test
Make sure Tensorleap’s integration works on your machine:
```
poetry run test
```

## 🔧 Install Tensorleap CLI
```
curl -s https://raw.githubusercontent.com/tensorleap/leap-cli/master/install.sh | bash
leap server install
```

## 🔐 Authenticate with Tensorleap

### 🔑 Generate a CLI Token
1.	[Request demo access](https://tensorleap.ai/request-demo/) from Tensorleap.
2.	After getting access, in the Tensorleap UI, open the menu (top-left).
3.	Click CLI TOKEN → YES, DISABLE MY OLD TOKEN AND GENERATE A NEW ONE.
4.	Copy the generated token to your clipboard
### 👨🏻‍💻 Login to Tensorleap
1. In the terminal, paste the generated token you copied, it should be in the following format: 
```
tensorleap auth login <API_KEY> <API_URL>
```
2. Run it to authenticate with Tensorlaep

## 🚢 Deploy Your Local Changes to the Project
```
leap projects push models/x3d.h5
```

## ✅ Validate & Run
1.	In Tensorleap’s UI, open the NETWORK tab.
2.	Click Code Integration → Validate Assets
Runs a small batch to confirm everything is configured correctly.
3.	Press Evaluate to process the full dataset.