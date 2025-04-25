# Contribution guidelines

### Welcome to *pyanno4rt*'s contribution guidelines

Thank you for considering contributing to *pyanno4rt*! As an open-source project, this repository lives from an engaged community with members like you.

We encourage and value all types of contributions regardless of experience level. Please see the [Table of Contents](#table-of-contents) for more information about how to help, or get in contact with us to discuss the format of your contribution!

> If you enjoy *pyanno4rt*, but just don't have time to contribute, that's fine. There are other easy ways to support the project and show your appreciation, which we would also be very happy about:
> - Star the project :star:
> - Cite the project in your work :pencil:
> - Share the project with others :earth_americas:

## Table of Contents :clipboard:

1) [Code of Conduct](#code-of-conduct)

2) [I Have a Question](#i-have-a-question)

3) [I Want To Contribute](#i-want-to-contribute)

4) [Styleguide](#styleguide)

## Code of Conduct ⚖️

Everyone contributing to the development of *pyanno4rt* is governed by our [Code of Conduct](CODE_OF_CONDUCT.md). If you also want to be a part of the project, we expect you to uphold this code. Please report unacceptable behavior to the developers.

## I Have a Question :question:

If you have a question about *pyanno4rt*, we recommend the following steps:

1. Search the project documentation on [Read the Docs](https://pyanno4rt.readthedocs.io/en/latest/)

2. Search the [Github Discussions](/discussions) and [Github Issues](/issues)

3. Search the internet

4. Open a new discussion topic
	> Please provide as much context as you can about what you're running into, e.g. project and platform versions. We will take care of your request as soon as possible.
	
5. Contact the developers

## I Want To Contribute :seedling:

> Legal note: when contributing to this project, you must agree that you have fully authored the content, that you have the necessary rights to the content and that the content you contribute may be provided under the project license!

Contributing to *pyanno4rt* is very easy - just make sure you read these guidelines carefully. It will not only facilitate the work for us maintainers, but also smooth out the experience for the whole community!

For contributions related to bug fixes, see [I Found a Bug](#i-found-a-bug) . For contributions related to ideas for the improvement of *pyanno4rt*, see [I Have a Suggestion](#i-have-a-suggestion). For contributions related to source codes, see [I Want To Add Code](#i-want-to-add-code).

### I Found a Bug

#### Before reporting a bug

- Make sure that you are using the latest version  of *pyanno4rt*

- Verify your bug to exclude any error on your side

- Check if the same bug has already been reported in the [bug tracker](issues?q=label%3Abug)

- Also, check the internet to see if users outside of the GitHub community have discussed the issue

- Collect information about the bug:
	- Stack trace (Traceback)
	- OS, platform and version (Windows, Linux, ...)
	- Version of the interpreter, compiler, SDK, runtime environment, package manager, depending on what seems relevant
	- Possibly your input and the output
	- Can you reliably reproduce the issue?

#### Reporting a bug

> You must never report security related issues, vulnerabilities or bugs including sensitive information to the issue tracker, or elsewhere in public. Instead sensitive bugs must be sent by email to the developers.

We use GitHub issues to track bugs and errors. If you run into an issue with *pyanno4rt*:

- Open a [new issue](/issues/new) without (!) calling or labelling the issue as a bug yet

- Explain the expected and the actual behavior

- Provide as much context as possible to ensure reproducibility of the issue, maybe including code (ideally: a minimum running example for testing)

After receiving your report, we will label the issue accordingly and reproduce it with your provided steps. If successful, we will try to fix the issue as quick as possible.

### I Have a Suggestion

#### Before submitting a suggestion

- Make sure that you are using the latest version of *pyanno4rt*

- Check the project documentation on [Read the Docs](https://pyanno4rt.readthedocs.io/en/latest/) for already existing functionality

- Search the [issues](/issues) for already submitted suggestions (if your suggestion is already present, add a comment instead of opening a new issue)

- Evaluate the fitness of your suggestion with the scope and aims of *pyanno4rt* (you can always make a strong case to convince us of the merits)


#### Submitting a suggestion

We use GitHub issues to track suggestions. If you want to make a suggestion for *pyanno4rt*:

- Select a clear and descriptive title for your suggestion

- Describe your idea with as many details as possible
	> Here, you may also add an explanation of the expected behavior compared to the current behavior, or include additional material like screenshots or code examples.

- Point out the relevance of your suggestion to our users

### I Want To Add Code

All code contributions should be in the form of [Pull Requests](/pulls).

Please follow the steps below to have your contribution considered by the maintainers:

1. Contact us to coordinate your code contribution

2. Follow all instructions in the [template](https://guides.github.com/activities/forking/)

3. Follow our [styleguide](#styleguide)

4. Make sure that your code passes all status checks

Pull requests that comply with these requirements will be checked carefully by us. If approved, we will include it in the next release!

## Styleguide :tophat:

During the development of *pyanno4rt*, we have followed a set of either established or self-imposed rules for code styling. These include, among others:

- Static code analysis with [Pylint](https://pylint.readthedocs.io/en/stable/)

- A maximum line length set to 79

- A base style for each file:
	```python 
	"""Label for the script."""

	# Author(s): Your Name, collaborator name(s)
	
	# %% External package import

	"Add external modules here, e.g. 'from numpy import array'"
	
	# %% Internal package import

	"Add internal modules here, e.g. 'from pyanno4rt.base import TreatmentPlan'"

	# %% Class/Function/Map definition

	"Add your code here"
	```

- A preference for 'from ... import ...' statements

- Alphabetical sorting of import statements (packages, modules, classes/functions), or grouping by headers

- Documentation using the [NumPy docstring style](https://numpydoc.readthedocs.io/en/latest/format.html)

- Naming of variables, functions, classes and other entities in your code aligned with the [PEP8 naming conventions](https://peps.python.org/pep-0008/#naming-conventions) (we also prefer human-readable names, i.e., 'plan_generator' instead of 'pln_gen')

- One-line comments for every step in the code

- Line breaks for method arguments (with functions, only the line length must be taken into account)

We have made a considerable effort to ensure consistency in the style of our source codes. Therefore, we recommend taking a look at the codes if anything is unclear.

**Thank you for being a good contributor!**

