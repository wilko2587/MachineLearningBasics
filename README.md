# README

## <u> Set Up </u>
Hi team.

We need to install some libraries now we're using pytorch, sklearn etc etc. The easiest way for us to all keep on the 
same page is with a requirements.txt file, which I've added in.

Go to your terminal in PyCharm (normally a tab in the bottom left corner of the screen), you run:

>pip install -r requirements.txt

And pip should install everything for you, so we're all using the same versions of every module. Easy.

Hope it works, if not just install stuff manually.

James


------
**Side note: how to make a requirements.txt file:** to generate (or overwrite) a requirements.txt file, you run
in terminal:

>pip freeze > requirements.txt

## <u> Work so far</u>

I've used the example from Nick's tutorial to set up a basic feed-forward neural network using PyTorch, in NN.py.
Currently, I'm calling this inside starter3.py to do some classifying as per the homework's instructions.