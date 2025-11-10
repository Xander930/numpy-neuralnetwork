<h1>Solving the XOR Problem</h1>

The XOR problem is a classic way of teaching small neural networks, it's how I learned to make them initially. Bu I'd never done it without libraries, which in my opinion are a poor way of teaching how NNs work. While true that havign a good command of up-to-date ML libraries is more of an employable skill than building NNs 'from scratch', I think it is useful to understand how each of the underlying principles looks in code by writing a network like this at least once.

<h2>The Beginnings of the Project</h2>

This was originally supposed to be a network for a seperate problem, as I recently aqcuired a copy of the book Neural Networks from Scratch and was interested in possibly creating the 3-class classifier shown there. But as I got further into writing the code I realized that I wanted to do a simpler, faster problem if I wanted to achieve my secondary goal of testing cross-language speed and seeing how well I could optimize python. So I changed my activation and loss functions and got to work.
It's far from efficient, but the current network works. It is far slower than one using libraries, but I will see how efficient I can get it as time goes on. I have a fairly powerful laptop, but I really want to see how light of a processor it can run on, as I want to spend some time working with lightweight classifiers running on ICs for robotics applications. Those are future projects, though. 


<h2>Results</h2>
After grappling with the Vanishing Gradient Problem for quite some time, I managed to get the network to train to 100% accuracy in 10,000 epochs. This is extremely slow for a neural network solving this simple of a problem, and it is also extremely unreliable, with the network sometimes requiring upwards of 25,000 epochs to fully train. This is a problem I will continue to work on as I make both reliability and efficiency improvements to this network going fowards.
