# Acknowledgements

This framework grew from my intent to build a principled multi-scale signal
analysis tool for personal use. Initial attempts used standard computational
methods, implemented by AI, to accelerate multi-window sweep aggregation; as
the desired feature set grew these approaches repeatedly broke down or could
not be aligned with my intent. Returning to first principles, I applied the
most basic form of reasoning about functions --- examining what an aggregation
function actually does at its simplest level --- and the divisibility lattice
fell out directly. A new computational method followed from it. The structure
is, of course, the discrete analogue of Riemann integration --- the grain is
the partition, and there is no limit to take --- a connection obvious to anyone
who has studied basic calculus.

My longstanding independent research and practice --- spanning more than two
decades --- of viewing integers through their prime coordinate structure, rooted
in formal training and deep study of number theory and abstract algebra, made
the p-adic connection immediate. The mathematical foundations were built by me
starting circa 2003 and solidified circa 2005–2008. I finished writing upfpy
circa 2008, but had working dirty code circa 2005 that I used to explore the
structure of integer factorization as a kind of vector space, built using web
tutorials and books, pre-AI:

https://github.com/rickhonda/upfpy

From there, as I built SignalForge through a number of prototypes, structural
benefits of the lattice revealed themselves in succession through a process of
me drawing on this mathematical background, with AI
assistance surfacing connections in adjacent territories, compounding
iteratively. I'm frankly kind of shocked that it implemented this way --- it
started with just trying to get a coarse fast track calculation for uniform
approximation surfaces.

Claude (Anthropic) and ChatGPT (OpenAI) served as research and implementation
staff, assisting with writing, formalization, and software implementation.

A fuller account is forthcoming.
