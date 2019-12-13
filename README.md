# IM_GreedyCELF

Jupyter Notebook source code for a [blog post](https://hautahi.com/im_greedycelf) comparing two key Influence Maximization algorithms - Greedy and CELF

## Installing igraph

I've often run into trouble getting the igraph package successfully plotting the igraph objects. This is quite a common problem, so the following is the installation method that I use, including my virtual environment setup:

- Create a new virtual environment with the following in bash: `$ mkvirtualenv -p pythonX environment_name`
- Install the relevant packages `$ pip install matplotlib pandas numpy python-igraph jupyterlab cairocffi`
- Add the environment to the jupyter notebook: `$ python -m ipykernel install --name=environment_name`

The above should successfully install the package with Python 2. But for Python 3 code, there's another very annoying step. We need to edit a particular igraph package source file. Its location will differ depending on your machine setup, but mine is located at `~/Envs/environment_name/lib/python3.7/site-packages/igraph/drawing/__init__.py`.

Within this file, there is a method `_repr_svw_()` that we need to edit. At approximately line 354, we need to replace `io.getvalue().encode("utf-8")` with `io.getvalue().decode("utf-8")`. The relevant github commit that details this change is [here](https://github.com/igraph/igraph/commit/037f89868190dd231f61d71ddeb3795ebc7e1274). There are also many stack overflow posts about this issue (see [here](https://stackoverflow.com/questions/57862852/python-igraph-bytes-object-has-no-attribute-encode-when-plotting-in-colab) and [here](https://stackoverflow.com/questions/12072093/python-igraph-plotting-not-available) for example).

Other Notes
- List the kernels available to jupyter with `$ jupyter kernelspec list`
- Remove a kernel with `$ jupyter kernelspec uninstall environment_name`
