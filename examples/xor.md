<h1>Solving the XOR Problem</h1>

The XOR problem is a classic way of teaching small neural networks, it's how I learned to make them initially. Bu I'd never done it without libraries, which in my opinion are a poor way of teaching how NNs work. While true that havign a good command of up-to-date ML libraries is more of an employable skill than building NNs 'from scratch', I think it is useful to understand how each of the underlying principles looks in code by writing a network like this at least once.

<h2>The Beginnings of the Project</h2>

This was originally supposed to be a network for a seperate problem, as I recently aqcuired a copy of the book Neural Networks from Scratch and was interested in possibly creating the 3-class classifier shown there. But as I got further into writing the code I realized that I wanted to do a simpler, faster problem if I wanted to achieve my secondary goal of testing cross-language speed and seeing how well I could optimize python. So I changed my activation and loss functions and got to work.
It's far from efficient, but the current network was trained to 100% accuracy in 1000 epochs. It could perhaps do it in less, I will see how efficient I can get it as time goes on. I have a fairly powerful laptop, but I really want to see how light of a processor it can run on, as I want to spend some time working with lightweight classifiers running on ICs for robotics applications. Those are future projects, though. 

<h2>Results</h2>

The network reached 100% accuracy in 10k epochs, with the loss graph attatched in the src/examples folder. I will get the exact time later on, as I am preparing to test various optimizations and compare it with similar networks in other languages. All in all, I learned a lot about the interal algorithms during the first part of this project, as well as version control as good practices, so I would consider it a resounding success!
