Contributing
============

We welcome contributions from the community! This guide will help you get started with contributing to GeomFuM.

Getting Started
---------------

Development Setup
~~~~~~~~~~~~~~~~~

The instructions in this section detail the step-by-step
process on setting up your development environment before
contribution.

We recommend using  `Git` for source control to allow collaboration. Typical interaction with the project involves using `git` to pull/push code and 
submitting bugs/feature requests to the `GeomFuM repository <https://github.com/3diglab/geomfum>`_.

Be sure to follow the Git installation and configuration instructions for your
respective operating system from the 
`official Git documentation <https://git-scm.com/book/en/v2/Getting-Started-Installing-Git>`_, 
before you follow along the next section of this documentation.


1. **Fork the repository**:

   Visit `https://github.com/3diglab/geomfum <https://github.com/3diglab/geomfum>`_ and click "Fork"

2. **Clone your fork**:

   .. code-block:: bash

       git clone https://github.com/<your-username>/geomfum.git
       cd geomfum

3. **Install in development mode**:

   .. code-block:: bash

       pip install -e .[test,,plotting-all]

Development Environment
~~~~~~~~~~~~~~~~~~~~~~~

We recommend using a virtual environment to manage dependencies:

.. code-block:: bash

    # Create virtual environment
    python -m venv geomfum_env
    source geomfum_env/bin/activate  # On Windows: geomfum_env\Scripts\activate
    
    # Install development dependencies
    pip install -e .[test,test-scripts,plotting-all]

Code Style
----------

We follow PEP 8 and use several tools to maintain code quality:

Formatting
~~~~~~~~~~

We use `ruff` for code formatting and linting. We recommend downloading the extension for your preferred IDE editor.


Documentation
~~~~~~~~~~~~~

The documentation is written in `rst` format and is located in the `docs/` directory.
The documentation is built using `sphinx` and is located in `https://3diglab.github.io/geomfum.github.io/ <https://3diglab.github.io/geomfum.github.io/>`_ directory.
The documentation is viewable in a web browser.

Writing Documentation
~~~~~~~~~~~~~~~~~~~~~

1. **API Documentation**: Add docstrings to all public functions
2. **Tutorials**: Create Jupyter notebooks in `notebooks/`
3. **Concept Documentation**: Add theoretical explanations in `docs/concepts/`

Example docstring:

.. code-block:: python

    def zoomout_refinement(
        functional_map: np.ndarray,
        mesh1: Mesh,
        mesh2: Mesh,
        iterations: int = 5
    ) -> np.ndarray:
        """Apply ZoomOut refinement to functional map.
        
        ZoomOut is a spectral upsampling technique that improves
        correspondence quality by iteratively refining the functional map.
        
        Parameters
        ----------
        functional_map : np.ndarray
            Input functional map matrix
        mesh1 : Mesh
            First mesh
        mesh2 : Mesh
            Second mesh
        iterations : int, optional
            Number of refinement iterations, by default 5
            
        Returns
        -------
        np.ndarray
            Refined functional map
            
        References
        ----------
        .. [1] Melzi, S., et al. "ZoomOut: Spectral Upsampling for Efficient
               Shape Correspondence." ACM TOG 38.6 (2019): 155.
        """
        pass

Pull Request Process
--------------------

1. **Create a feature branch**:

   .. code-block:: bash

      git checkout -b feature/your-feature-name

2. **Make your changes**:

   - Write code following our style guidelines
   - Add tests for new functionality
   - Update documentation

3. **Run tests and checks**:

   .. code-block:: bash

      # Run tests
      pytest

4. **Commit your changes**:

   .. code-block:: bash

      git add .
      git commit -m "Add feature: brief description"

      # Use conventional commit format:
      # feat: add new functional map refinement
      # fix: resolve memory leak in laplacian computation
      # docs: update installation instructions

5. **Push and create the branch on the remote**:

   .. code-block:: bash

      git push origin feature/your-feature-name

6. **Create the Pull Request**:

   - Use the PR template
   - Describe your changes clearly
   - Link related issues
   - Request reviews from maintainers

7. **Wait for review**:

   - Address feedback from maintainers
   - Make changes as requested
   - Re-request review when ready
   - Ensure all checks pass before merging
   - Merge when approved

Code Review
-----------

Review Process
~~~~~~~~~~~~~~

1. **Automated checks** must pass:

   - Tests
   - Code style
   - Documentation build
   - Type checking

2. **Manual review** by maintainers:

   - Code quality
   - Algorithm correctness
   - Performance considerations
   - Documentation quality

3. **Address feedback**:

   - Respond to review comments
   - Make requested changes
   - Re-request review when ready

Review Guidelines
~~~~~~~~~~~~~~~~~

When reviewing code:

1. **Check correctness**: Is the algorithm implemented correctly?
2. **Check performance**: Are there performance issues?
3. **Check style**: Does the code follow our conventions?
4. **Check documentation**: Is the documentation clear and complete?
5. **Check tests**: Are there adequate tests?

Areas for Contribution
----------------------

We welcome contributions in these areas:

Algorithms
~~~~~~~~~~

- New functional map algorithms
- Improved refinement techniques
- Novel shape descriptors
- Deep learning approaches

Performance
~~~~~~~~~~~

- GPU acceleration
- Parallel processing
- Memory optimization
- Sparse matrix operations

Documentation
~~~~~~~~~~~~~

- Tutorials and examples
- API documentation
- Theoretical explanations
- Performance benchmarks

Testing
~~~~~~~

- Unit tests
- Integration tests
- Performance tests
- Documentation tests

Tools and Utilities
~~~~~~~~~~~~~~~~~~~

- Visualization tools
- Data processing utilities
- Evaluation metrics
- Benchmarking tools

Getting Help
------------

If you need help:

1. **Check existing issues**: Search for similar problems
2. **Join Discord**: `https://discord.gg/THHku2ckJs <https://discord.gg/THHku2ckJs>`_
3. **Create an issue**: For bugs or feature requests
4. **Ask questions**: In GitHub Discussions

Community Guidelines
--------------------

We are committed to providing a welcoming and inclusive environment:

1. **Be respectful**: Treat everyone with respect
2. **Be helpful**: Help others learn and contribute
3. **Be patient**: Everyone learns at their own pace
4. **Be constructive**: Provide constructive feedback

Code of Conduct
~~~~~~~~~~~~~~~

We follow the Contributor Covenant Code of Conduct. Please read it at:
`https://www.contributor-covenant.org/version/2/0/code_of_conduct.html <https://www.contributor-covenant.org/version/2/0/code_of_conduct.>`_

Recognition
-----------

Contributors are recognized in:

1. **GitHub contributors list**
2. **Documentation acknowledgments**
3. **Release notes**
4. **Academic citations** (for significant contributions)

Thank you for contributing to GeomFuM! 🎉