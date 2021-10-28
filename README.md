# Naïve Tsetlin Machine Implementation

This is a tryout implementation of the Tsetlin Machine [^1].

A Tsetlin Machine (TM) is machine learning (ML) approach that uses discrete logic to learn first-order-logic clauses which determine the output (the output, however, is not FOL, but based on all clauses voting, much like in a random forest).

## Benefits
According to the authors, it offers better explainability than some other ML approaches. This is because the learned clauses can be simply read (if X1 and X2 then Y). A caveat here is that (1) the "Y" is not the prediction of the complete algorithm, but rather "an argument for Y", that is, one vote of potentially many, many of whom might contradict each other. When you have a few hundred clauses, this will make interpretability harder, and no better than that of logistic regression, support vector machines, or random forests, and probably much worse than decision trees (but it outperforms those, to be honest). There's a tradeoff as the number of clauses is a hyperparameter. Increasing it reduces interpretability, but lowering it makes TM much less competitive.

Its authors also benchmark TM and would like to see it outperform other algorithms, but in my view it's still somewhat early for real state-of-the-art (SOTA) benchmarks. One big promise is, however, that, for the accuracy that it achieves this can be done with much fewer and simpler calculation steps than SOTA algorithms. This is particularly interesting for hardware settings where large-scale floating-point calculations are not or cannot be optimized due to cost or power efficiency reasons (or you just to get better ML results with existing hardware) [^2][^3]. In my view, this is also very revelant for the current generation of highly efficient multi-core ARM CPUs, where users can potentially achieve perfectly decent results with standard consumer hardware, even without GPU or dedicated NN ASIC support (such as Apple's M1 SoC family).

## Working Principle
TMs use finite state automata called Tsetlin Automata (TA), each of which turns on/off one single bit of the input based on its current state (or state range, rather). The turning on/off output is called the action of the TA. It's input bit is called an input literal (it might be a negated version of the original input bit or not). State transitions take place based on a reward, moving one state toward more extreme/polarized states for the current action for each reinforcing reward, and one state back towards the middle and eventually opposite action for each deterring reward (avoiding the term "negative" here, as it's really a 0).

The number of clauses to create is a hyperparameter (even positive number) which influences the fidelity of the model as well as the interpretability, as stated above. Each clause is a conjugation with two terms per input bit, one as-is, and one negated. If there are two inputs, a clause would look like this: x₁ ∧ x₂ ∧ ¬x₁ ∧ ¬x₂. As explained earlier, some of these terms might be ignored (=turned off). If the TAs for x₂ and ¬x₁, for example, learn to return action 0 (=turn off input), the learned clause will be: x₁ ∧ ¬x₂. There needs to be an even number of clauses because half of them will vote for output 1, and half of them will vote for output 0 (this is separate from the negation of inputs).

For learning, the reward feedback for each individual TA is determined from the global target value (1 or 0), its clause's polarity (does it vote for 1 or 0), the clause result (did the clause the TA belongs to actually fire or not), the literal (was the TA's xᵢ/¬xᵢ equal to 1 or 0), and the TA's action (1=turn off literal or 0=turn off literal).

There are two types of feedback: For positive-polarity clauses, Feedback I has the purpose of making the clause result approach the target, while Feedback II is meant to suppress the clause result 1. These are swapped for negative polarity clauses. Both use a stochastic element to make it possible that the generated clauses are actually different. I must say that the way the two types of feedback interact is somewhat interwoven, and there are several sub-aspects that warrant to be examined separately. Let's just say that Feedback I tends to have clauses grow larger (include more literals) for correct answers, and shrink them (exclude literals) for incorrect answers, while Feedback II focuses on "antagonistic" clauses only (that is, clauses whose polarity is the opposite of the target) and shrinks them if they contribute to create false positives.

Apparently, how the feedback is to be presented cleanly, is still in flux. In [^1] the authors used tables, while in [^4] they changed the structure. I find that the latter describes the intention behind the probabilities a little better and creates fewer "meaningless" table cells, while dealing with negative clauses is clearer in the former.

The voting design entails that (num clauses hyperparameter permitting) the TM will learn duplicate clauses. These make interpretation harder, which is why in [^4], the authors introduce weighting of clause votes, with the goal of achieving comparable results with a lower number of clauses.

There are many proposed or implemented exensions to TMs. The original paper [^1] gives an overview, [^4] introduces convolutional TMs. Some exensions and applications I want to look at but have not gotten around to yet are TM-based regression, layered TMs and TMs in NLP.

# Implementation
'tsetlin.py` is a naïve implementation in Python, solely for purposes of playing with and understanding the algorithm. You are free to use it but it comes with no warranties (Apache 2.0 License). However, I recommend to use the original implementation [^5]. The implementation wastes pretty much all of the opportunities for a high efficiency algorithms (bits are represented by integers, for example). Please note that the original authors

The `Automaton` class describes a single TA, including its feedback transition.

The `Machine` class implements the prediction and learning algorithm, including the feedback, and weighting (optional). I've also added a `clauses_str` method for getting a  readable overview of the current clauses.

The body of the file contains some tests with noisy XOR. When playing around, I did not find weighting to help particularly well with this use case. I also never managed to get down to the optimal number of four clauses for a non-noisy XOR case, which I expected to be able to reach. Also, random seed seems to play a considerable role for the performance of low-clause-count TMs. However, I'm not sure if this is to be expected or is due to bugs in my implementation. If you have any hints (or questions) feel free to post them in an issue. :)

[^1]: Granmo, O.-C. (2018). The Tsetlin Machine--A Game Theoretic Bandit Driven Approach to Optimal Pattern Recognition with Propositional Logic. *ArXiv Preprint [ArXiv:1804.01508](https://arxiv.org/abs/1804.01508)*.

[^2]: [Jie Lei's tutorial on YouTube, also includes logic simulation (in the hardware/FPGA sense)](https://www.youtube.com/watch?v=XzWSPo7GF94&list=PLQTEHj1nqgNmBHtiw5l5cOs986WUKp8FZ)

[^3]: [Adrian Wheeldon's presentation looks into power consumption as well](https://www.youtube.com/watch?v=TaspuovmSR8)

[^4]: Granmo, O.-C., Glimsdal, S., Jiao, L., Goodwin, M., Omlin, C. W., & Berge, G. T. (2019). The convolutional Tsetlin machine. *ArXiv Preprint [ArXiv:1905.09688](https://arxiv.org/abs/1905.09688)*.

[^5]: [TsetlinMachine at GitHub](https://github.com/cair/TsetlinMachine)
