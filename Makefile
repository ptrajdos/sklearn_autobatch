ROOTDIR=$(realpath $(dir $(firstword $(MAKEFILE_LIST))))


SRCDIR=${ROOTDIR}/sklearn_autobatch
TESTDIR=${ROOTDIR}/tests
COVDIR=${ROOTDIR}/htmlcov_p
COVERAGERC=${ROOTDIR}/.coveragerc
INSTALL_LOG_FILE=${ROOTDIR}/install.log
VENV_SUBDIR=${ROOTDIR}/venv
COVERAGERC=${ROOTDIR}/.coveragerc
DOCS_DIR=${ROOTDIR}/docs
TOXDIR=${ROOTDIR}/.tox

COVERAGE = coverage
UNITTEST_PARALLEL = unittest-parallel
PDOC= pdoc3
PYTHON=python
SYSPYTHON=python
PIP=pip
PYTEST=pytest
TOX=tox
VENV_OPTIONS=

LOGDIR=${ROOTDIR}/testlogs
LOGFILE=${LOGDIR}/`date +'%y-%m-%d_%H-%M-%S'`.log


PYTHON_VERSION=3.9

ifeq ($(OS),Windows_NT)
	ACTIVATE:=. ${VENV_SUBDIR}/Scripts/activate
else
	ACTIVATE:=. ${VENV_SUBDIR}/bin/activate
endif

.PHONY: all clean test docs

clean: clean_pypackages clean_venv clean_tox
	@echo "Cleaning up build artifacts, virtual environments, and test logs..."

clean_pypackages:
	rm -rf pypackages

clean_venv:
	rm -rf ${VENV_SUBDIR}

clean_tox:
	rm -rf ${TOXDIR}
venv:
	${SYSPYTHON} -m venv --upgrade-deps ${VENV_OPTIONS} ${VENV_SUBDIR}
	${ACTIVATE}; ${PYTHON} -m ${PIP} install wheel setuptools pypackages
	

pypackages: venv
	${ACTIVATE}; ${PYTHON} -m ${PIP} install -e ${ROOTDIR}[dev] --prefer-binary --log ${INSTALL_LOG_FILE}
	touch $@

test: pypackages
	mkdir -p ${LOGDIR}  
	${ACTIVATE}; ${COVERAGE} run --branch  --source=${SRCDIR} -m unittest discover -p '*_test.py' -v -s ${TESTDIR} 2>&1 |tee -a ${LOGFILE}
	${ACTIVATE}; ${COVERAGE} html --show-contexts


test_parallel: pypackages
	mkdir -p ${COVDIR} ${LOGDIR}
	${ACTIVATE}; ${UNITTEST_PARALLEL} --class-fixtures -v -t ${ROOTDIR} -s ${TESTDIR} -p '*_test.py' --coverage --coverage-rcfile ./.coveragerc --coverage-source ${SRCDIR} --coverage-html ${COVDIR}  2>&1 |tee -a ${LOGFILE}

docs: pypackages
	${ACTIVATE}; $(PDOC) --force --html ${SRCDIR} --output-dir ${DOCS_DIR}

profile: pypackages
	
	${ACTIVATE}; ${PYTEST} -n auto --cov-report=html --cov=${SRCDIR} --profile ${TESTDIR}

tox_check: pypackages
	${ACTIVATE}; ${TOX}
