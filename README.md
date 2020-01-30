# Adversarial Debiasing

## Authors:

Rochelle Choenni (10999949, rmvk97@gmail.com)
<br />
Maximilian Filtenborg (11042729, max.filtenborg@gmail.com)
<br />
Victoria Foing (11773391, vickyfoing@gmail.com)
<br />
Gaurav Kudva (12205583, gauravkudva2@gmail.com)

## TA

Leon Lang (leon.lang@student.uva.nl)

## Installation

We have a conda environment.yml file specifying the required packages:

    $ conda env create -f environment.yml

And activate the environment:

    $ conda activate fact
   
Update your environment later (after activating it):

    $ conda env update --file environment.yml

Update your env with the requirements.txt (packages only in pip):

    $ pip install -r requirements.txt

Update the env after installing a package:

    $ conda env export --from-history > environment.yml

If you add a package from pip, add it to the requirements.txt

