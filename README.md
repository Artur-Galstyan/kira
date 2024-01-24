# That's right. I'm _**Kira**_ ✍️

_Kira_ is a transformer built with JAX and
[Equinox](https://github.com/patrick-kidger/equinox), which is IMHO the best
neural network library built on top of JAX out there, because of its simple and
elegant design.

The role of _Kira_ is to serve as a baseline transformer implementation, which
is very easy to understand and extend.

_Kira_ now allows for KV caching. It also allows you to interpolate
between Multi-Query Attention (MQA) and "regular" Multi-Head Attention (MHA).
This is different from other MHA implementations, where most of the time you can
only set the number of heads and that's it. _Kira_ offers more flexibility in
that regard.

These features of _Kira_ (the interpolation between MQA and MHA and the RoPE
embeddings) will soon be integrated in the main Equinox repository at which
point _Kira_'s MHA implementation will be replaced with the built-in Equinox's
MHA.

_Kira_ can also be used as an encoder. Simply pass `mask=None` when you call _Kira_ 
and there will be no masking in the MHA (i.e. making it an encoder).

---

To get started with _Kira_, you can either install it with

```
pip3 install kira_llm
```

or simply clone the repository and cherry-pick what you need.

## Contributing

To contribute to this project, you'll need to fork this repository and run

```
poetry install
```

which will install all of the dependencies. Then, simply make your changes and
start a pull request. The philosophy of this repository is to be simple and
understandable.
