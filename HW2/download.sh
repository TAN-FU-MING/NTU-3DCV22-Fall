if [ ! -d data ]
then
	wget https://drive.google.com/u/0/uc\?export\=download\&confirm\=qrVw\&id\=1GrCpYJFc8IZM_Uiisq6e8UxwVMFvr4AJ -O data.zip  
	unzip data.zip && rm data.zip
fi
