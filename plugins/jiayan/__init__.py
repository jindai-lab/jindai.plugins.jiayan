"""甲言插件"""

from jindai import PipelineStage, Plugin, expand_path
from jindai.helpers import safe_import
from jindai.models import Paragraph


class JiayanPlugin(Plugin):
    """甲言插件"""

    def __init__(self, app, **config) -> None:
        safe_import('kenlm', 'https://github.com/kpu/kenlm/archive/master.zip')
        jiayan = safe_import('jiayan')
        load_lm = jiayan.load_lm

        class JiayanWordCut(PipelineStage):
            """文言文分词"""

            tokenizer = None

            def __init__(self):
                super().__init__()
                if JiayanWordCut.tokenizer is None:
                    JiayanWordCut.tokenizer = jiayan.CharHMMTokenizer(
                        load_lm(expand_path('models_data/jiayan.klm')))

            def resolve(self, paragraph: Paragraph):
                paragraph.tokens = list(
                    JiayanWordCut.tokenizer.tokenize(paragraph.content))

        class JiayanPOSTagger(PipelineStage):
            """标注文言文词性"""

            postagger = None

            def __init__(self):
                super().__init__()
                if JiayanPOSTagger.postagger is None:
                    JiayanPOSTagger.postagger = jiayan.CRFPOSTagger()
                    JiayanPOSTagger.postagger.load(
                        expand_path('models_data/pos_model'))

            def resolve(self, paragraph: Paragraph):
                paragraph.pos = JiayanPOSTagger.postagger.postag(
                    paragraph.tokens)

        self.register_pipelines(locals())
        super().__init__(app, **config)
