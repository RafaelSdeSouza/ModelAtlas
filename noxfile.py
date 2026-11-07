"""Repeat the review checks in isolated environments."""

import nox

nox.options.sessions = ["lint", "tests", "docs", "build"]
TOOLING = "requirements/tooling.txt"
FIGURES = "requirements/figures-py312.txt"


@nox.session(python="3.12")
def lint(session):
    """Check lint and annotated core APIs."""
    session.install("-c", FIGURES, "-r", TOOLING, "-e", ".")
    session.run(
        "ruff", "check", "src", "tests", "examples", "scripts", "docs/conf.py", "docs/_ext", "noxfile.py"
    )
    session.run("mypy")


@nox.session(python=["3.10", "3.11", "3.12", "3.13", "3.14"])
def tests(session):
    """Run the numerical and plotting suite headlessly."""
    session.install("-c", TOOLING, "-e", ".[test]")
    session.env["MPLBACKEND"] = "Agg"
    session.run("pytest", "--cov=modelatlas", *session.posargs)


@nox.session(python="3.12")
def docs(session):
    """Execute the gallery and verify local links and figures."""
    session.install("-c", FIGURES, "-e", ".[docs,examples]")
    session.env.update({"MPLBACKEND": "Agg", "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1"})
    session.run("sphinx-build", "-E", "-a", "-W", "-n", "-b", "html", "docs", "docs/_build/html")
    session.run("python", "scripts/check_docs.py", "docs/_build/html")


@nox.session(python="3.12")
def build(session):
    """Build both distribution formats and check their metadata."""
    session.install("-r", TOOLING)
    session.run("python", "-m", "build")
    session.run("twine", "check", "--strict", "dist/*")
