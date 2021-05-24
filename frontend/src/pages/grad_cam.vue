<template>
    <div class="page">
        <div class="menu">
            <div class="item">
                <div class="title">model</div>
                <el-select class="value" size="small" v-model="params.model" @change="getLayers">
                    <el-option v-for="model in models" :key="model" :value='model'/>
                </el-select>
                </div>
            <div class="item">
                <div class="title">input</div>
                <el-select class="value" size="small" v-model="params.input" @change="config">
                    <el-option value='static/images/cat_dog.png'/>
                    <el-option value='static/images/spider.png'/>
                    <el-option value='static/images/snake.jpg'/>
                </el-select>
            </div>
            <div class="item">
                <div class="title">layer</div>
                <el-select class="value" size="small" v-model="params.layer" @change="update">
                    <el-option v-for="layer in layers" :value="layer.index" :key="layer.index">{{layer.index}} - {{layer.name}} / {{layer.layer}}</el-option>
                </el-select>
            </div>
            <div class="item"><div class="title">target</div><el-input class="value" size="small" v-model="params.target"  @change="update"/></div>
        </div>
        <div class="network">
            <div class="input"><img :src="params.input" crossorigin='anonymous'/></div>
            <div class="sliency">
                <div class="image-wrapper"><img class="image" :src="res.grayscale" crossorigin='anonymous'/><div class="caption">Grayscale</div></div>
                <div class="image-wrapper"><img class="image" :src="res.colorful" crossorigin='anonymous'/><div class="caption">Grad CAM</div></div>
                <div class="image-wrapper"><img class="image" :src="res.on_image" crossorigin='anonymous'/><div class="caption">Grad CAM * Image</div></div>
            </div>
            <div class="sliency">
                <div class="image-wrapper"><img class="image" :src="res.guided_saliecy" crossorigin='anonymous'/><div class="caption">Guided Gradient</div></div>
                <div class="image-wrapper"><img class="image" :src="res.guided_grad_cam" crossorigin='anonymous'/><div class="caption">Guided Grad CAM</div></div>
            </div>
        </div>
    </div>
</template>

<script>
export default {
    name: 'saliency',
    data() {
        return {
            models: [],
            params: {
                model: 'alexnet',
                input: 'static/images/cat_dog.png',
                layer: 11,
                target: 243
            },
            layers: {},
            res: {}
        };
    },
    created() {
        this.config()
        this.update()
    },
    sockets: {
        models(data) {
            this.models = data
        },
        layers(data) {
            this.layers = data
        },
        response_gradcam(data) {
            this.res = data
        }
    },
    methods: {
        config() {
            this.$socket.emit('get_models')
            this.getLayers()
            this.update()
        },
        getLayers() {
            this.$socket.emit('get_layers', { model: this.params.model })
        },
        update() {
            this.$socket.emit("gradcam", this.params);
        }
    }
};
</script>

<style rel="stylesheet/scss" lang="scss" scoped>
.page {
    .network {
        display: flex;
        flex-flow: row;

        align-items: center;
        justify-items: center;

        .input {
            flex: 0 0 auto;
        }

        .sliency {
            flex: 1 1 auto;
            display: flex;
            flex-flow: column;
            align-items: center;
            justify-items: center;
        }
    }
}

</style>
