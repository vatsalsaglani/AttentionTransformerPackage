from setuptools import setup


setup(
	name='Attention-Transformer',
    version = 'v0.0.1',
    url = 'https://github.com/vatsalsaglani/Attention-Transformer-.git',
    description='Sequence to Sequence Multi Head Attention Transformer package with more to come train Seq2Seq models.',
	author = 'Vatsal Saglani',
    license = 'MIT',
    install_requires = ['torch', 'torchvision', 'tqdm', 'torchtext', 'numpy', 'scipy', 'scikit-learn'],
    packages = ['AttentionTransformer'],
	zip_safe = False

)