import unittest
import os
import sys
from subprocess import call

test_dir = os.path.dirname(os.path.abspath(__file__))
examples_dir = os.path.join(test_dir, '..', '..', 'examples')


class TestExamples(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        pass

    @classmethod
    def tearDownClass(self):
        pass

    def test_that_examples_run(self):
        cwd = os.getcwd()
        os.chdir(examples_dir)
        example_files = [
            f
            for f in os.listdir(examples_dir)
            if os.path.isfile(os.path.join(examples_dir, f))
            and f.endswith(".py")
            and not f.startswith("test")
        ]
        flag = 0
        failed_examples = []
        for f in example_files:
            tmp_flag = call([sys.executable, os.path.join(examples_dir, f)])
            print(f, tmp_flag)
            if tmp_flag == 1:
                failed_examples.append(f)
                flag = 1
        os.chdir(cwd)
        if len(failed_examples) > 0:
            print("failed examples: {0}".format(failed_examples))
        self.assertEqual(flag, 0)


if __name__ == "__main__":
    unittest.main()
