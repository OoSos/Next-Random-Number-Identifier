Contributing
============

Development Guidelines
--------------------

This project follows several best practices for Python development:

* Type checking with MyPy
* Code formatting with Black
* Import ordering with isort
* Linting with Flake8
* Testing with pytest

Understanding the Architecture
----------------------------

Before making changes, please review these important architectural documents:

* :doc:`Architecture Documentation <architecture>`
* Component Interaction Diagram
* Data Flow Diagram
* Prediction Sequence Diagram

These documents will help you understand:

* The overall system design
* How components interact
* Data flow through the system
* The prediction process

Contributing Process
------------------

1. Fork the repository
2. Create your feature branch (``git checkout -b feature/amazing-feature``)
3. Run the tests and linting checks (``pytest`` and ``flake8``)
4. Commit your changes (``git commit -m 'Add some amazing feature'``)
5. Push to the branch (``git push origin feature/amazing-feature``)
6. Open a Pull Request

Code Quality
-----------

All contributions must pass:

* Unit tests (``pytest``)
* Type checking (``mypy``)
* Code formatting (``black``)
* Import sorting (``isort``)
* Linting (``flake8``)