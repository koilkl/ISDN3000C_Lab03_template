# ISDN3000C_Lab03: RDK X5 AI Vision Challenge Template

This is the official template repository for the Lab03 of ISDN3000C.

This repository contains the necessary files to run a simple AI image classification task on your RDK X5.

## Repository Contents

*   `classify.py`: A Python script that uses the **PyTorch** library to load a pre-trained AI model (`ResNet18`), analyze the `sample_image.png`, and print the classification result.
*   `requirements.txt`: libraries to install before running `classify.py`.
*   `classify_batch.py`: for advanced task.
*   `sample_image.png`: A sample image of a  used as input for the AI model.
*   `classify_batch.py`: A advanced script to handle batch image
*   `result.csv`:A csv file include the information of match result. 
## Instructions

1.  **Fork this Repository**: Create your own copy of this repository on GitHub.
2.  **Clone it to your RDK**: Follow the assignment instructions to install `git` and clone **your forked repository** onto the RDK X5.
3.  **Install Dependencies**: As per the assignment, ensure you have installed the required Python libraries on your RDK:
    ```bash
    pip install -r requirements.txt
    ```
4.  **Run the Script**: Navigate into the repository directory on your RDK and execute the script:
    ```bash
    python classify.py
    python classify_batch.py yourfoldername
    (Do not use /yourfoldername,because it's not a valid directory name)
    ```
5.  **Use the Output**: The script will print the AI's prediction. Use this output to complete the final steps of the assignment (creating the webpage and submitting your work).
