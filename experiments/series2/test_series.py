"""Small regression tests for the new numerical and recovery paths."""
import json, tempfile, unittest
from pathlib import Path
import mlx.core as mx
import numpy as np
from mlx_lm.utils import load_model
import convert_inkling as converter
from test_converter import make_source
from model_tools import Statistics, load_reference, expert_modules, ObservedSwitch, case_arrays
from clipping import cpu_search, apply_clipping
from common import tensor_id, sha_file

class SeriesTests(unittest.TestCase):
    def setUp(self):mx.set_default_device(mx.cpu);mx.random.seed(27)
    def tearDown(self):converter.ACTIVE_RECIPE=None;converter.ACTIVE_SEARCH=None

    def test_single_letter_answer_boundary_excludes_template_footer(self):
        class Tokenizer:
            def apply_chat_template(self,messages,**kwargs):
                return '<user>'+messages[0]['content'][0]['text']+'<assistant_content>'+messages[-1]['content']+'<end_sampling>'
            def encode(self,text,**kwargs):return list(text.encode())
        case={'question':'Return the letter a','answer':'a'}
        ids,_,start,count=case_arrays(case,Path('.'),Tokenizer(),{})
        values=np.asarray(ids[0]).tolist()
        self.assertTrue(bytes(values[:start]).decode().endswith('<assistant_content>'))
        self.assertEqual(bytes(values[start:]).decode(),'a')
        self.assertEqual(count,1)

    def test_nvfp4_global_scaling_and_no_clipping_keep_full_precision(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);make_source(root);model,_=load_model(root,lazy=True,strict=True)
            module=next(m for _,m in expert_modules(model) if m.mode=='nvfp4')
            w=converter.reconstruct_expert(module,0)
            self.assertEqual(w.dtype,mx.float32)
            idx=np.zeros((w.shape[0],w.shape[1]//64),np.uint8)
            clipped=apply_clipping(w,idx,64)
            self.assertEqual(clipped.dtype,mx.float32)
            self.assertTrue(mx.array_equal(w,clipped).item())
            for actual,expected in zip(mx.quantize(clipped,group_size=64,bits=2),mx.quantize(w,group_size=64,bits=2)):
                self.assertTrue(mx.array_equal(actual,expected).item())

    def test_reference_text_image_and_loop_statistics(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);make_source(root);stats=Statistics()
            model,config=load_reference(root,stats)
            stats.begin_case();out=model(mx.array([[1,2,3]*22]));mx.eval(out)
            self.assertTrue(mx.all(mx.isfinite(out)).item())
            for name,count in stats.counts.items():
                visits=3 if '.layers.1.' in name or '.layers.2.' in name else 1
                self.assertEqual(int(count.sum()),66*visits)
                self.assertEqual(len(stats.visits[name]),visits)
            stats.begin_case();vision=model(mx.array([[1,127,2]]),pixel_values=[mx.ones((3,4,4))]);mx.eval(vision)
            self.assertTrue(mx.all(mx.isfinite(vision)).item())
            before={k:v.copy() for k,v in stats.counts.items()};stats.enabled=False
            out=model(mx.array([[1,2]]));mx.eval(out)
            for name in before:np.testing.assert_array_equal(stats.counts[name],before[name])

    def test_unsorted_topk_statistics_duplicate_inputs_correctly(self):
        stat=Statistics();x=mx.array([[[[[1.,2.]]],[[[3.,4.]]]]])
        idx=mx.array([[[0,1],[1,0]]])
        stat.observe('projection',x,idx,2)
        np.testing.assert_array_equal(stat.counts['projection'],[2,2])
        np.testing.assert_array_equal(stat.sums['projection'],[[10,20],[10,20]])

    def test_clipping_never_worsens_its_own_objective(self):
        w=mx.random.normal((8,128)).astype(mx.bfloat16);w[0,0]=8
        stat=np.arange(1,129,dtype=np.float32)/64
        maps,metrics=cpu_search(w,stat)
        for group in (32,64,128):
            for bits in (2,4):
                self.assertLessEqual(metrics[f'C_g{group}_b{bits}'],metrics[f'R_g{group}_b{bits}']+1e-6)
                self.assertLessEqual(metrics[f'W_g{group}_b{bits}'],metrics[f'RW_g{group}_b{bits}']+1e-6)
                for method in ('C','W'):
                    clipped=apply_clipping(w,maps[f'{method}_g{group}_b{bits}'],group)
                    self.assertTrue(mx.all(mx.isfinite(clipped)).item())

    def test_mixed_precision_export_and_interrupted_resume(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);source=root/'source';source.mkdir();make_source(source)
            source_hashes={str(p.relative_to(source)):sha_file(p) for p in source.rglob('*') if p.is_file()}
            model,_=load_model(source,lazy=True,strict=True);search=root/'search';search.mkdir()
            for path,module in expert_modules(model):
                for e in range(module.num_experts):
                    w=converter.reconstruct_expert(module,e)
                    maps,_=cpu_search(w,alphas=[1.,.95,.90])
                    lookup=np.array([0,4,6],np.uint8)
                    np.savez(search/(tensor_id(path,e)+'.npz'),**{k:lookup[v] for k,v in maps.items()})
            del model
            recipe={'id':'C64-test','method':'C','group_size':64,'embedding_and_head_bits':4,
                'attention_bits':8,'expert_layers_at_4_bits':[3]}
            original=converter.repack;calls=[]
            def interrupted(*args,**kwargs):
                calls.append(kwargs['path'])
                if len(calls)==2:raise RuntimeError('deliberate interruption')
                return original(*args,**kwargs)
            converter.repack=interrupted
            try:
                with self.assertRaisesRegex(RuntimeError,'deliberate interruption'):
                    converter.convert(source,root/'out',64,recipe=recipe,search_dir=search,checkpoint_dir=root/'checkpoints')
            finally:converter.repack=original
            self.assertTrue(list((root/'checkpoints').glob('*.safetensors')))
            report=converter.convert(source,root/'out',64,recipe=recipe,search_dir=search,checkpoint_dir=root/'checkpoints')
            self.assertEqual(report['status'],'converted_and_structurally_verified')
            loaded,_=load_model(root/'out',lazy=True,strict=True)
            for path,module in expert_modules(loaded):self.assertEqual(module.bits,4 if '.layers.3.' in path else 2)
            for path,module in expert_modules(loaded):
                self.assertEqual(module.scales.dtype,mx.bfloat16 if '.layers.3.' in path else mx.float32)
            self.assertEqual(loaded.lm_head.bits,4)
            loaded.set_dtype(mx.float32)
            out=loaded(mx.array([[1,127,2]]),pixel_values=[mx.ones((3,4,4))]);mx.eval(out)
            self.assertTrue(mx.all(mx.isfinite(out)).item())
            self.assertEqual(source_hashes,{str(p.relative_to(source)):sha_file(p) for p in source.rglob('*') if p.is_file()})

if __name__=='__main__':unittest.main(verbosity=2)
