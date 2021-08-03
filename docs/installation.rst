************
Installation
************

.. include:: ../README.rst
   :start-after: install-start
   :end-before: install-end

Optional Requirements
*********************

During installation, GloMPO only installs core requirements. However, many of its most useful abilities are optional extensions with other package requirements.

To install GloMPO with extra requirements:

.. code-block:: bash

   pip install .[cma, checkpointing]

To install GloMPO with all (publicly available) extra requirements:

.. code-block:: bash

   pip install .[all]

The full list of optional requirements:

.. include:: ../extra_requirements.txt
   :start-after: tab-start
   :end-before: tab-end

Tests
*****

.. include:: ../README.rst
   :start-after: test-start
   :end-before: test-end

.. note::

   If your tests fail please raise an issue as described in :ref:`Raising Issues`.
