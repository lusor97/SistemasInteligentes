****************HOW TO EXECUTE****************
- 	First of all, you have to open "PacMan_ReinforcementLearning.ipynb" using, 
	for example, Google Colab ( https://colab.research.google.com ).
-	After doing that, run the first cell (which install everything you need).
-	Open the "Files" tab and click on "Upload". You will have to select "deep_Q.py",
	"duel_Q.py", "main.py", "pacman.py" and "replay_buffer.py" for making it work.
	If you also want to be able to continue from the point we trained to, upload also
	"saved_DQN.h5" and "saved_DDQN.h5".
-	Once you have uploaded everything, you have to know what the next cell does: 
	"!python main.py" executes the "main.py" file with the parameters you entered.
	"-n" is the algorithm and NN you want to use (change it between "DDQN" or "DQN").
	"-m" is the mode (change between "train" (for training the NN) and "test" (for 
	getting results)).
	"-l" loads the NN saved files (it doesn't matter what you introduce here because 
	we made some changes in "main.py" and it will just take "saved_DDQN.h5" or 
	"saved_DQN.h5" depending on the parameter "-n", for making things easier for you).
	"-s" creates a folder where the testing data will be saved (as a game video of the
	AI playing).
-	You can now run the cell and it will work as asked in the parameters.
	If you want to change the number of frames for training, just change "NUM_FRAME" 
	value in "main.py".
___________________________________________________________________________________________

*******************LG Team:*******************
-	Javier Verón
-	Jorge Chueca
-	Luis Ríos
___________________________________________________________________________________________

This project counts with a license agreed to the terms of the license MIT 2.0